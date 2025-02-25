package models.generative;

import data.Dataset;
import org.ejml.simple.SimpleMatrix;

public class QDAClassifier extends GenerativeClassifier {

    public SimpleMatrix[] covMatrices;

    public QDAClassifier(int nClasses) {
        super(nClasses);
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
        covMatrices = new SimpleMatrix[nClasses];
        for (int i = 0; i < nClasses; i++) {
            SimpleMatrix ones = new SimpleMatrix(classPredictor[i].getNumCols(), 1).plus(1);
            SimpleMatrix bcMeans = ones.mult(meansVectors.getRow(i)).transpose();
            SimpleMatrix meanDiff = classPredictor[i].minus(bcMeans);
            covMatrices[i] = meanDiff.mult(meanDiff.transpose());
            covMatrices[i] = covMatrices[i].scale(1.0 / (labelCount[i] - 1));
        }
    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        SimpleMatrix scores = new SimpleMatrix(data.getNumRows(), nClasses);
        SimpleMatrix meansVectors = new SimpleMatrix(means);
        for (int i = 0; i < nClasses; i++) {
            for (int j = 0; j < data.getNumRows(); j++) {
                SimpleMatrix meanVector = meansVectors.getRow(i);
                SimpleMatrix x = data.getRow(j);
                SimpleMatrix quadTerm = x.mult(covMatrices[i].invert()).mult(x.transpose()).scale(-0.5);
                SimpleMatrix predictorTerm = x.mult(covMatrices[i].invert()).mult(meanVector.transpose());
                SimpleMatrix biasTerm = meanVector.mult(covMatrices[i].invert()).mult(meanVector.transpose()).scale(-0.5).plus(Math.log(priors[i]));
                SimpleMatrix score = quadTerm.plus(predictorTerm).plus(biasTerm);
                scores.set(j, i, score.get(0, 0));
            }
        }
        return scores.toArray2();
    }
}
