/*
Authors: Kenneth Tran <one@kentran.net>
License: BSD 3 clause
 */

#pragma once

#include <vector>

namespace TimeSeries {
    typedef std::string string;
    typedef std::vector<float> FloatVec;
    typedef std::vector<string> StrVec;

    class StreamingAnomalyDetector {
    public:
        FloatVec Thresholds; // Note that Thresholds can be updated live without breaking the service

        StreamingAnomalyDetector(int windowSize, FloatVec& thresholds);
        StreamingAnomalyDetector(const char* modelFile);
        void Save(const char* modelFile);
        int Predict(float value, float& trend, float& score);

    private:
        size_t windowSize_;
        FloatVec coefs_;
        FloatVec buffer_;
        void Init();
    };

    struct Window {
        int RecordId;
        string Begin;
        string End;
        float Score;
        Window(int recordId, string& begin, string& end, float score);
    };

    std::vector<Window> DetectAnomalies(const char* dataFile, int windowSize, int levels=1, bool hasHeader=true, char separator=',', const char* modelFile = nullptr);
}
