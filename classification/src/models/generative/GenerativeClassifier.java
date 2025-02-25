package models.generative;

import data.Dataset;
import models.ClassificationModel;
import org.ejml.simple.SimpleMatrix;

public abstract class GenerativeClassifier extends ClassificationModel {

    public double[] priors;
    public double[][] means;
    public int nPredictors;

    public GenerativeClassifier(int nClasses) {
        priors = new double[nClasses];
        means = new double[nClasses][];
        this.nClasses = nClasses;
    }

    public void calculatePriorsAndMeans(double[][] labels, double[][] predictors) {
        int nPredictors = predictors[0].length;
        this.nPredictors = nPredictors;
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
                    break;
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

    public abstract void fit(Dataset dataset);

    public abstract void calculateCovMatrix(double[][] labels, double[][] predictors);

    public abstract double[][] predict(SimpleMatrix data);

    @Override
    public void logOdds() {

    }
}
