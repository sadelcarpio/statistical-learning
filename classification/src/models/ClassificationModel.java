package models;

import data.Dataset;
import metrics.Metric;
import org.ejml.simple.SimpleMatrix;

import java.util.HashMap;
import java.util.Map;

public abstract class ClassificationModel {
    public int nClasses;

    public Map<String, Double> evaluate(Dataset dataset, Metric[] metrics) {
        double[][] predictors = dataset.getPredictors();
        double[][] yTrue = dataset.getLabels();
        double[][] yPred = predict(new SimpleMatrix(predictors));
        Map<String, Double> metricsMap = new HashMap<>();
        for (Metric metric: metrics) {
            metricsMap.put(metric.getMetricName(), metric.calculate(yTrue, yPred));
        }
        return metricsMap;
    }

    public abstract double[][] predict(SimpleMatrix data);
}
