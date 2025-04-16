package models.generative;

import data.Dataset;
import models.ClassificationModel;
import org.ejml.simple.SimpleMatrix;

/**
 * Base class for Generative Classification models.
 */
public abstract class GenerativeClassifier extends ClassificationModel {

    /**
     * Prior probabilities of belonging to a class. Shape {@code [K]}
     */
    public double[] priors;
    /**
     * Means of each predictor across each class. Shape {@code [K][p]}
     */
    public double[][] means;
    /**
     * Number of labels per class. Shape {@code [K]}
     */
    public double[] labelCount;
    /**
     * Array of K matrices, each one of them contains the predictors that belong to each class.
     */
    public SimpleMatrix[] classPredictor;

    /**
     * a.k.a {@code n}, number of examples
     */
    public int totalRows;

    /**
     * a.k.a {@code p}, number of predictors
     */
    public int nPredictors;

    public GenerativeClassifier(int nClasses) {
        priors = new double[nClasses];
        means = new double[nClasses][];
        classPredictor = new SimpleMatrix[nClasses];
        this.nClasses = nClasses;
    }

    /**
     * Calculates the prior probabilities ({@code num_obs_k / total_obs}) and the means per predictor, per class
     * @param labels one-hot encoded labels
     * @param predictors predictors
     */
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

    /**
     * Fits the model by calculating the priors, means and Covariate Matrix. The posterior probability is:
     * {@code p(k | x) = prior[k] * p(x | k) / sum(prior[k] * p(x | k))}, where {@code p(x | k) ~ N(means, cov) }
     * @param dataset {@link Dataset} object, containing predictors and labels
     */
    @Override
    public void fit(Dataset dataset) {
        double[][] labels = dataset.getLabels();
        double[][] predictors = dataset.getPredictors();
        calculatePriorsAndMeans(labels, predictors);
        calculateCovMatrix(labels, predictors);
    }

    /**
     * Separates the predictors according to the class they belong
     * @param labels one-hot encoded labels
     * @param predictors numeric predictors
     */
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

    /**
     * Calculate the Covariance Matrix.
     * @param labels one-hot encoded labels
     * @param predictors predictors numeric predictors
     */
    public abstract void calculateCovMatrix(double[][] labels, double[][] predictors);

    /**
     * Predicts which class a given set of predictors belong to
     * @param data predictors matrix of shape {@code [n][p]}
     * @return labels matrix of shape {@code [n][K]}
     */
    @Override
    public abstract double[][] predict(SimpleMatrix data);
}
