package models.generative;

import data.Dataset;
import models.ClassificationModel;

import java.util.ArrayList;
import java.util.Map;

public abstract class GenerativeClassifier extends ClassificationModel {
    @Override
    public void fit(Dataset dataset) {

    }

    @Override
    public Map<String, Double> evaluate(Dataset dataset) {
        return Map.of();
    }

    @Override
    public ArrayList<Integer> predict(ArrayList<ArrayList<Float>> data) {
        return null;
    }

    @Override
    public void log_odds() {

    }
}
