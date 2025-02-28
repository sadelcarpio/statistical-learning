package models.generative;

import org.ejml.simple.SimpleMatrix;

public class GaussianNaiveBayes extends GenerativeClassifier {

    public double[][] variances;

    public GaussianNaiveBayes(int nClasses) {
        super(nClasses);
    }

    private double gaussianDist(double x, double mu, double variance) {
        return (Math.exp(-0.5 * Math.pow((x - mu), 2) / variance)) / (Math.sqrt(2 * Math.PI * variance));
    }

    @Override
    public void calculateCovMatrix(double[][] labels, double[][] predictors) {
        getClassPredictors(labels, predictors);
        variances = new double[nClasses][nPredictors];
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < nPredictors; j++) {
                variances[i][j] = classPredictor[i].getRow(j).minus(means[i][j]).elementPower(2).elementSum() / (labelCount[i] - 1);
            }
        }
    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        int totalRowsInference = data.getNumRows();
        double[][] scores = new double[totalRowsInference][nClasses];
        double[] evidence = new double[totalRowsInference];
        for (int i = 0; i < totalRowsInference; i++) {
            evidence[i] = 0;
            for (int j = 0; j < nClasses; j++) {
                double likelihood = 1;
                for (int k = 0; k < nPredictors; k++) {
                    likelihood *= gaussianDist(data.get(i, k), means[j][k], variances[j][k]);
                }
                scores[i][j] = priors[j] * likelihood;
                evidence[i] += scores[i][j];
            }
        }

        for (int i = 0; i < totalRowsInference; i++) {
            for (int j = 0; j < nClasses; j++) {
                scores[i][j] /= evidence[i];
            }
        }
        return scores;
    }
}
