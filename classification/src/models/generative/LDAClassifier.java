package models.generative;

import org.ejml.simple.SimpleMatrix;

public class LDAClassifier extends GenerativeClassifier {

    public SimpleMatrix covMatrix;

    public LDAClassifier(int nClasses) {
        super(nClasses);
    }

    public double[][] predict(SimpleMatrix data) {
        int totalRowsInference = data.getNumRows();
        SimpleMatrix scores = new SimpleMatrix(totalRowsInference, nClasses);
        SimpleMatrix meansVectors = new SimpleMatrix(means);
        for (int i = 0; i < nClasses; i++) {
            SimpleMatrix meanVector = meansVectors.getRow(i);
            SimpleMatrix predictorTerm = data.mult(covMatrix.invert()).mult(meanVector.transpose());
            SimpleMatrix ones = new SimpleMatrix(totalRowsInference, 1).plus(1);
            SimpleMatrix biasTerm = meanVector.mult(covMatrix.invert()).mult(meanVector.transpose()).scale(-0.5).plus(Math.log(priors[i]));
            SimpleMatrix bcBiasTerm = ones.mult(biasTerm);
            SimpleMatrix score = predictorTerm.plus(bcBiasTerm);
            for (int j = 0; j < totalRowsInference; j++) {
                scores.set(j, i, score.get(j));
            }
        }
        return scores.toArray2();
    }

    @Override
    public void calculateCovMatrix(double[][] labels, double[][] predictors) {
        getClassPredictors(labels, predictors);
        SimpleMatrix meansVectors = new SimpleMatrix(means);
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
