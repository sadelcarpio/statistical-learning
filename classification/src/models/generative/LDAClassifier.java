package models.generative;

import data.Dataset;
import org.ejml.simple.SimpleMatrix;

public class LDAClassifier extends GenerativeClassifier {

    public SimpleMatrix covMatrix;

    public LDAClassifier(int nClasses) {
        super(nClasses);
    }

    @Override
    public void fit(Dataset dataset) {
        double[][] labels = dataset.getLabels();
        double[][] predictors = dataset.getPredictors();
        calculatePriorsAndMeans(labels, predictors);
        calculateCovMatrix(labels, predictors);
    }

    public double[][] predict(SimpleMatrix data) {
        SimpleMatrix scores = new SimpleMatrix(data.getNumRows(), nClasses);
        SimpleMatrix meansVectors = new SimpleMatrix(means);
        for (int i = 0; i < nClasses; i++) {
            SimpleMatrix meanVector = meansVectors.getRow(i);
            SimpleMatrix predictorTerm = data.mult(covMatrix.invert()).mult(meanVector.transpose());
            SimpleMatrix ones = new SimpleMatrix(data.getNumRows(), 1).plus(1);
            SimpleMatrix biasTerm = meanVector.mult(covMatrix.invert()).mult(meanVector.transpose()).scale(-0.5).plus(Math.log(priors[i]));
            SimpleMatrix bcBiasTerm = ones.mult(biasTerm);
            SimpleMatrix score = predictorTerm.plus(bcBiasTerm);
            for (int j = 0; j < data.getNumRows(); j++) {
                scores.set(j, i, score.get(j));
            }
        }
        return scores.toArray2();
    }

    @Override
    public void calculateCovMatrix(double[][] labels, double[][] predictors) {
        int totalRows = labels.length;
        SimpleMatrix meansVectors = new SimpleMatrix(means);
        SimpleMatrix[] classPredictor = new SimpleMatrix[nClasses];
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
        covMatrix = new SimpleMatrix(nPredictors, nPredictors);
        for (int i = 0; i < nClasses; i++) {
            SimpleMatrix ones = new SimpleMatrix(classPredictor[i].getNumCols(), 1).plus(1);
            SimpleMatrix bcMeans = ones.mult(meansVectors.getRow(i)).transpose();
            SimpleMatrix meanDiff = classPredictor[i].minus(bcMeans);
            covMatrix = covMatrix.plus(meanDiff.mult(meanDiff.transpose()));
        }
        covMatrix = covMatrix.scale(1.0 / (totalRows - nClasses));
    }
}
