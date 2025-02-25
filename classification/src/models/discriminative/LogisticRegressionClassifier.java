package models.discriminative;

import data.Dataset;
import models.ClassificationModel;
import org.ejml.simple.SimpleMatrix;

public class LogisticRegressionClassifier extends ClassificationModel {

    public SimpleMatrix beta;

    public LogisticRegressionClassifier(int nClasses) {
        this.nClasses = nClasses;
    }

    public void fit(Dataset dataset) {
        SimpleMatrix predictorsMatrix = new SimpleMatrix(dataset.getPredictors());
        SimpleMatrix ones = new SimpleMatrix(predictorsMatrix.getNumRows(), 1).plus(1);
        SimpleMatrix designMatrix = ones.concatColumns(predictorsMatrix);
        beta = SimpleMatrix.random(designMatrix.getNumCols(), nClasses);
        var yTrue = new SimpleMatrix(dataset.getLabels());
        optimize(designMatrix, yTrue);
    }

    private void optimize(SimpleMatrix designMatrix, SimpleMatrix yTrue) {
        for (int i = 0; i < 1000; i++) {
            SimpleMatrix z = designMatrix.mult(beta);
            SimpleMatrix yHat = softMax(z);
            double loss = -yTrue.elementMult(yHat.elementLog()).elementSum();
            SimpleMatrix gradBeta = designMatrix.transpose().mult(yHat.minus(yTrue));
            beta = beta.minus(gradBeta.scale(0.01));
        }
    }

    private SimpleMatrix softMax(SimpleMatrix matrix) {
        SimpleMatrix ones = new SimpleMatrix(matrix.getNumCols(), matrix.getNumCols()).plus(1);
        return matrix.elementExp().elementDiv(matrix.elementExp().mult(ones));
    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        SimpleMatrix ones = new SimpleMatrix(data.getNumRows(), 1).plus(1);
        SimpleMatrix designMatrix = ones.concatColumns(data);
        SimpleMatrix z = designMatrix.mult(beta);
        return softMax(z).toArray2();
    }

    @Override
    public void logOdds() {

    }
}
