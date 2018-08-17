
class CrossValidator():
    """Temporal cross-validator class.

    This class performs temporal (causal) cross-validation similar to the
    approach in https://robjhyndman.com/papers/cv-wp.pdf.

    :param forecaster: Forecaster.
    :type forecaster: A forecaster class
    :param val_frac: Fraction of data to be used for validation per fold.
    :type val_frac: float
    :param n_folds: Number of temporal folds.
    :type n_folds: int
    :param loss: The kind of loss used for evaluating the forecaster on folds.
    :type loss: string

    """
    def __init__(self,
                 forecaster,
                 fold_generator,
                 loss='normal_log_likelihood',
                 metrics=['smape', 'pinball_loss']):
        """Initialize properties."""

        # Forecaster properties
        self.forecaster = forecaster
        self.targets = None
        self.n_samples = 1000

        # Cross-validation properties
        self.fold_generator = fold_generator  # Must be a generator
        self.loss = loss
        self.metrics = metrics
        self.prediction_samples = None

    def evaluate(self, targets=None, verbose=1):
        """Evaluate forecaster."""
        lag = self.forecaster.lag
        horizon = self.forecaster.horizon

        # Forecaster fitting and prediction parameters
        self.targets = targets

        # Set up the metrics dictionary also containing the main loss
        percentiles = np.linspace(0, 100, 101)
        percentile_names = ['p' + str(x) for x in percentiles]
        metrics = pd.DataFrame(
            columns=[self.loss, ] + self.metrics + percentile_names
        )

        for i, data_train in enumerate(self.fold_generator):
            # Set up the forecaster
            forecaster = self.forecaster
            forecaster._is_fitted = False  # Make sure we refit the forecaster
            t0 = time.time()

            # Quietly fit the forecaster to this fold's training set
            forecaster.fit(
                data_train,
                targets=self.targets,
                verbose=0  # Fit in silence
            )

            # Depending on the horizon, we make multiple predictions on the
            # test set and need to create those input output pairs
            j = 0
            inputs = []
            while (j + 1) * horizon + lag <= data_train.shape[1]:
                tmp = []
                for time_series in data_train:
                    tmp.append(time_series[j * horizon:j * horizon + lag, :])
                inputs.append(np.array(tmp))
                j += 1

            # Time series values to be forecasted
            n_horizon = (data_train.shape[1] - lag) // horizon
            if self.targets:
                data_pred = data_train[
                    :, lag:lag + n_horizon * horizon, self.targets
                ]
            else:
                data_pred = data_train[:, lag:lag + n_horizon * horizon, :]

            # Make predictions for each of the input chunks
            prediction_samples = []
            for input_data in inputs:
                samples = forecaster.predict(
                    input_data,
                    n_samples=self.n_samples
                )
                prediction_samples.append(samples)
            prediction_samples = np.concatenate(prediction_samples, axis=2)

            # Update the loss for this fold
            metrics_append = {}
            metrics_append[self.loss] = getattr(custom_metrics, self.loss)(
                prediction_samples,
                data_pred
            )

            # Update other performance metrics for this fold
            for metric in self.metrics:
                func = getattr(custom_metrics, metric)
                metrics_append[metric] = func(
                    prediction_samples,
                    data_pred
                )

            # Update coverage metrics for this fold
            for perc in percentiles:
                metrics_append['p' + str(perc)] = custom_metrics.coverage(
                    prediction_samples,
                    data_pred
                )
            metrics = metrics.append(metrics_append, ignore_index=True)

            # Update the user on the validation status
            duration = round(time.time() - t0)
            if verbose > 0:
                print("Validation fold {} took {} s.".format(i, duration))

        # Clean up the metrics table
        avg = pd.DataFrame(metrics.mean()).T
        avg.index = ['avg.']
        std = pd.DataFrame(metrics.std()).T
        std.index = ['std.']
        metrics = pd.concat([metrics, avg, std])
        metrics = metrics.round(2)
        metrics.index.name = 'fold'

        # Store predictions in case they are needed for plotting
        self.prediction_samples = prediction_samples

        return metrics
