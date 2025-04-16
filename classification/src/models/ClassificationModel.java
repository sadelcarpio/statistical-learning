package models;

import data.Dataset;
import metrics.Metric;
import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.Map;

/**
 * Base model for classification
 */
public abstract class ClassificationModel {
    public int nClasses;

    /**
     * Evaluate a model on a given set of metrics
     *
     * @param dataset {@link Dataset} object, containing predictors and labels
     * @param metrics Array of {@link Metric} objects
     * @return Map of metric names and their values
     */
    public Map<String, Double> evaluate(Dataset dataset, Metric[] metrics) {
        double[][] predictors = dataset.getPredictors();
        double[][] yTrue = dataset.getLabels();
        double[][] yPred = predict(new SimpleMatrix(predictors));
        Map<String, Double> metricsMap = new HashMap<>();
        for (Metric metric : metrics) {
            metricsMap.put(metric.getMetricName(), metric.calculate(yTrue, yPred));
        }
        return metricsMap;
    }

    /**
     * Predicts the labels for a set of predictors
     *
     * @param data predictors matrix of shape {@code [n][p]}
     * @return labels matrix of shape {@code [n][K]}
     */
    public abstract double[][] predict(SimpleMatrix data);

    /**
     * Updates the model parameters to fit the dataset it's being trained on.
     * @param dataset {@link Dataset} object, containing predictors and labels
     */
    public abstract void fit(Dataset dataset);
}
