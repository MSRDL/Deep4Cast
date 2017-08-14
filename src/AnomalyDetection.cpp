/*
WARNING: this is prototype code. It has not been tested.
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#include <algorithm>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"
#include "StreamingAnomalyDetector.h"


namespace TimeSeries {
    using namespace std;
    using namespace rapidjson;

    inline void CheckArg(bool condition, const char* message) {
        if (!condition)
            throw invalid_argument(message);
    }

    FloatVec ComputeCoefs(int w) {
        FloatVec coefs(4);
        coefs[0] = w * (w - 1) * (2 * w - 1) / 6; // a[1,1]
        coefs[1] = w * (w - 1) / 2; // a[1,2] = a[2,1]
        coefs[2] = w; // a[2,2]
        coefs[3] = 1 / (coefs[0] * coefs[2] - coefs[1] * coefs[1]);
        return coefs;
    }

    float ComputeScore(const FloatVec& coefs, const FloatVec& values, float& trend) {
        assert(coefs.size() == 4);

        double b1, b2;
        const float* py = &values[0];
        for (int j = 0; j < values.size(); ++j, ++py) {
            //The computations here can be optimized using SSE if needed
            float y = *py;
            b1 += j * y;
            b2 += y;
        }
        trend = (coefs[2] * b1 - coefs[1] * b2) * coefs[3];
        float intercept = (b1 - coefs[0] * trend) / coefs[1];

        float residual = 0;
        py = &values[0];
        for (int j = 0; j < values.size(); ++j, ++py) {
            float y = *py;
            float deviation = trend * j + intercept - y;
            residual += deviation * deviation;
        }
        return sqrt(residual / values.size());
    }

    StreamingAnomalyDetector::StreamingAnomalyDetector(int windowSize, FloatVec& thresholds)
            : windowSize_(windowSize), Thresholds(move(thresholds))
    {
        Init();
    }

    StreamingAnomalyDetector::StreamingAnomalyDetector(const char* modelFile) {
        FILE* pFile = fopen(modelFile, "r");
        char buffer[256];
        FileReadStream is(pFile, buffer, sizeof(buffer));
        Document doc;
        doc.ParseStream(is);

        windowSize_ = (doc["WindowSize"]).GetInt();
        const Value& thresholds = doc["Thresholds"];
        for (auto itr = thresholds.Begin(); itr != thresholds.End(); ++itr)
            Thresholds.push_back((float)itr->GetDouble());

        Init();
    }

    void StreamingAnomalyDetector::Save(const char* modelFile) {
        StringBuffer s;
        Writer<StringBuffer> writer(s);

        writer.StartObject();
        writer.String("WindowSize");
        writer.Int(windowSize_);
        writer.String("Thresholds");
        writer.StartArray();
        for (float threshold:Thresholds)
            writer.Double(threshold);
        writer.EndArray();
        writer.EndObject();

        ofstream(modelFile) << s.GetString();
    }

    void StreamingAnomalyDetector::Init() {
        CheckArg(Thresholds.size() >= 1, "Thresholds must contain at least 1 value");
        float cur = Thresholds[0];
        for (auto iter = Thresholds.begin() + 1; iter != Thresholds.end(); ++iter) {
            CheckArg(*iter > cur, "Thresholds must be increasing");
            cur = *iter;
        }
        coefs_ = ComputeCoefs(windowSize_);
        buffer_.resize(windowSize_);
        for (float& x : buffer_)
            x = numeric_limits<float >::quiet_NaN();
    }

    // Binary search to find the level
    inline int FindLevel(const FloatVec& thresholds, float score) {
        int l = 0;
        int r = thresholds.size();
        int m;
        while (l < r) {
            m = (l+r) / 2;
            assert(m < r);
            float threshold = thresholds[m];
            if (threshold < score)
                l = m+1;
            else
                r = m;
        }
        return score < thresholds[m] ? m : m + 1;
    }

    int StreamingAnomalyDetector::Predict(float value, float &trend, float &score) {
        memcpy(&buffer_[0], &buffer_[1], (windowSize_-1) * sizeof(float));
        buffer_.back() = value;
        return FindLevel(Thresholds, ComputeScore(coefs_, buffer_, trend));
    }

    class CsvReader {
        ifstream stream_;
        const char delim_;

    public:
        CsvReader(const string& filename, char delim = ',', bool hasHeader=true)
                : stream_(ifstream(filename)), delim_(delim)
        {
            if (hasHeader)
            {
                string header;
                if (!getline(stream_,header))
                    throw invalid_argument("Empty file");
            }
        }

        bool GetLine(vector<string>& tokens) {
            string line;
            if (!getline(stream_, line))
                return false;

            stringstream ss(line);
            tokens.clear();
            string item;
            while (getline(ss, item, delim_))
                tokens.push_back(move(item));

            return true;
        }
    };

    Window::Window(int recordId, string& begin, string& end, float score): RecordId(recordId), Begin(move(begin)), End(move(end)), Score(score) {}

    vector<Window> DetectAnomalies(const char*dataFile, int windowSize, int levels, bool hasHeader, char separator, const char* modelFile) {
        const clock_t start = clock();
        CheckArg(windowSize > 0, "Window size must be positive");
        CheckArg(levels > 0, "Number of alert levels must be positive");
        CsvReader reader(dataFile, separator, hasHeader);
        const int stride = max(1, windowSize / 4); // stride can be as small as 1 but we don't need stride too-small, for better efficiency
        FloatVec coefs = ComputeCoefs(windowSize);

        StrVec timestamps(windowSize);
        FloatVec values(windowSize);
        StrVec tokens;
        for (int i = 1; i < windowSize; ++i) {
            CheckArg(reader.GetLine(tokens), "Not enough data points or window size is too big");
            CheckArg(tokens.size() == 2, "Invalid data format: data must contain two columns, one for timestamp and one for counter value");
            timestamps[i] = move(tokens[0]);
            values[i] = stof(tokens[1]);
        }
        float trend;
        vector<Window> windows {Window(0, timestamps[0], timestamps.back(), ComputeScore(coefs, values, trend))};

        int counter = 0;
        int subCounter = 0;
        StrVec newTimeStamps(stride);
        FloatVec newValues(stride);
        while(reader.GetLine(tokens)) {
            newTimeStamps[subCounter] = move(tokens[0]);
            newValues[subCounter] = stof(tokens[1]);
            if ((++subCounter) == stride) {
                memmove(&timestamps[0], &newTimeStamps[0], stride * sizeof(string));
                memmove(&values[0], &newValues[0], stride * sizeof(float));
                counter += subCounter;
                windows.push_back(Window(counter, timestamps[0], timestamps.back(), ComputeScore(coefs, values, trend)));
                subCounter = 0;
            }
        }

        // Sort windows by score in descending order
        sort(windows.begin(), windows.end(), [](Window& a, Window& b) {return a.Score > b.Score;});

        // Filtering overlapping windows
        vector<Window> selected {move(windows[0])};
        vector<int> diffs {0}; // differences between adjacent ranked windows
        auto cur = windows.begin() + 1;
        auto lim = windows.end();
        while (cur != lim) {
            CHECK_OVERLAPPING:
                for (Window& sel:selected) {
                    if (abs(cur->RecordId - sel.RecordId) < windowSize) {
                        if (++cur == lim)
                            goto COMPUTE_THRESHOLDS;
                        else
                            goto CHECK_OVERLAPPING;
                    }
                }
            diffs.push_back(selected.back().Score - cur->Score);
            selected.push_back(move(*cur));
            ++cur;
        }

        COMPUTE_THRESHOLDS:
            vector<int> topJumps(diffs.size()); // top jumps are the window indices of where the biggest jumps happen
            iota(topJumps.begin(), topJumps.end(), 0);
            sort(topJumps.begin(), topJumps.end(), [&](int i, int j) {return diffs[i] > diffs[j];});
            sort(topJumps.begin(), topJumps.begin() + levels); // after figuring out the top jumps, reorder them by the anomaly index

            FloatVec thresholds(levels);
            for (int il = 0; il < levels; ++il) {
                int iw = topJumps[il];
                thresholds[il] = (selected[iw -1].Score - selected[iw].Score) / 2;
            }

        // Print to console output
        int iw = 0;
        for (int il = 0; il < levels; ++il) {
            int iwLim = topJumps[il];
            for (; iw < iwLim; ++iw) {
                Window& window = selected[iw];
                cout << setw(45) << (window.Begin + " - " + window.End) << window.Score << endl;
            }
            cout << "--------------- Threshold level " << levels - il << ": " << thresholds[il] << " ---------------" << endl;

        }

        if (modelFile != nullptr)
            StreamingAnomalyDetector(windowSize, thresholds).Save(modelFile);

        cout << "Time elapsed: " << double(clock() - start) / CLOCKS_PER_SEC << endl;
        return selected;
    }
}

#include <boost/python.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(TimeSeries)
{
    def("DetectAnomalies", TimeSeries::DetectAnomalies);
}
