package models.generative;

import data.Dataset;
import models.ClassificationModel;
import org.ejml.simple.SimpleMatrix;

public abstract class GenerativeClassifier extends ClassificationModel {

    public double[] priors;
    public double[][] means;
    public double[] labelCount;
    public SimpleMatrix[] classPredictor;
    public int totalRows;
    public int nPredictors;

    public GenerativeClassifier(int nClasses) {
        priors = new double[nClasses];
        means = new double[nClasses][];
        classPredictor = new SimpleMatrix[nClasses];
        this.nClasses = nClasses;
    }

    public void calculatePriorsAndMeans(double[][] labels, double[][] predictors) {
        this.nPredictors = predictors[0].length;
        int numRows = labels.length;

        labelCount = new double[nClasses];
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

    public void fit(Dataset dataset) {
        double[][] labels = dataset.getLabels();
        double[][] predictors = dataset.getPredictors();
        calculatePriorsAndMeans(labels, predictors);
        calculateCovMatrix(labels, predictors);
    }

    protected void getClassPredictors(double[][] labels, double[][] predictors) {
        totalRows = labels.length;
        for (int i = 0; i < totalRows; i++) {
            for (int j = 0; j < nClasses; j++) {
                if (labels[i][j] != 0) {
                    if (classPredictor[j] == null) {
                        classPredictor[j] = new SimpleMatrix(predictors[i]);
                    } else {
                        classPredictor[j] = classPredictor[j].concatColumns(new SimpleMatrix(predictors[i]));
                    }
                    break;
                }
            }
        }
    }

    public abstract void calculateCovMatrix(double[][] labels, double[][] predictors);

    public abstract double[][] predict(SimpleMatrix data);
}
