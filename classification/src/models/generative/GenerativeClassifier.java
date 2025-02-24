package models.generative;

import data.Dataset;
import metrics.Metric;
import models.ClassificationModel;
import org.ejml.simple.SimpleMatrix;

import java.util.Map;

public abstract class GenerativeClassifier extends ClassificationModel {
    public void fit(Dataset dataset) {

    }

    @Override
    public Map<String, Double> evaluate(Dataset dataset, Metric[] metrics) {
        return Map.of();
    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        return null;
    }

    @Override
    public void logOdds() {

    }
}
