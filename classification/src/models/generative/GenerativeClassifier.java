package models.generative;

import data.Dataset;
import models.ClassificationModel;
import org.ejml.simple.SimpleMatrix;

import java.util.Arrays;

public abstract class GenerativeClassifier extends ClassificationModel {

    double[] priors;
    double[][] means;

    public GenerativeClassifier(int nClasses) {
        priors = new double[nClasses];
        means = new double[nClasses][];
        this.nClasses = nClasses;
    }

    public void fit(Dataset dataset) {
    }

    public void calculatePriorsAndMeans(double[][] labels, double[][] predictors) {
        int nPredictors = predictors[0].length;
        int numRows = labels.length;

        double[] labelCount = new double[nClasses];
        means = new double[nClasses][nPredictors];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < nClasses; j++) {
                labelCount[j] += labels[i][j];
                if (labels[i][j] != 0) {
                    for (int k = 0; k < nPredictors; k++) {
                        means[j][k] += predictors[i][k];
                    }
                }
            }
        }
        for (int i = 0; i < nClasses; i++) {
            priors[i] = labelCount[i] / numRows;
            for (int j = 0; j < nPredictors; j++) {
                means[i][j] = means[i][j] / labelCount[i];
            }
        }
    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        return null;
    }

    @Override
    public void logOdds() {

    }
}
