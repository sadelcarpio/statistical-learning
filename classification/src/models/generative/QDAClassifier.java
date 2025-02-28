package models.generative;

import org.ejml.simple.SimpleMatrix;

public class QDAClassifier extends GenerativeClassifier {

    public SimpleMatrix[] covMatrices;

    public QDAClassifier(int nClasses) {
        super(nClasses);
    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        int totalRowsInference = data.getNumRows();
        SimpleMatrix scores = new SimpleMatrix(totalRowsInference, nClasses);
        SimpleMatrix meansVectors = new SimpleMatrix(means);
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < totalRowsInference; j++) {
                SimpleMatrix meanVector = meansVectors.getRow(i);
                SimpleMatrix x = data.getRow(j);
                SimpleMatrix quadTerm = x.mult(covMatrices[i].invert()).mult(x.transpose()).scale(-0.5);
                SimpleMatrix predictorTerm = x.mult(covMatrices[i].invert()).mult(meanVector.transpose());
                SimpleMatrix biasTerm = meanVector.mult(covMatrices[i].invert()).mult(meanVector.transpose())
                        .scale(-0.5).plus(Math.log(covMatrices[i].determinant()) * -0.5).plus(Math.log(priors[i]));
                SimpleMatrix score = quadTerm.plus(predictorTerm).plus(biasTerm);
                scores.set(j, i, score.get(0, 0));
            }
        }
        return scores.toArray2();
    }

    @Override
    public void calculateCovMatrix(double[][] labels, double[][] predictors) {
        getClassPredictors(labels, predictors);
        SimpleMatrix meansVectors = new SimpleMatrix(means);
        covMatrices = new SimpleMatrix[nClasses];
        for (int i = 0; i < nClasses; i++) {
            SimpleMatrix ones = new SimpleMatrix(classPredictor[i].getNumCols(), 1).plus(1);
            SimpleMatrix bcMeans = ones.mult(meansVectors.getRow(i)).transpose();
            SimpleMatrix meanDiff = classPredictor[i].minus(bcMeans);
            covMatrices[i] = meanDiff.mult(meanDiff.transpose());
            covMatrices[i] = covMatrices[i].scale(1.0 / (labelCount[i] - 1));
        }
    }
}
