package models;

import data.Dataset;

import java.util.ArrayList;
import java.util.Map;

public abstract class ClassificationModel {
    public int nClasses;

    public abstract void fit(Dataset dataset);

    public abstract Map<String, Double> evaluate(Dataset dataset);

    public abstract ArrayList<Integer> predict(ArrayList<ArrayList<Float>> data);

    public abstract void log_odds();
}
