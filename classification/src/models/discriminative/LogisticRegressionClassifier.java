package models.discriminative;

import data.Dataset;
import models.ClassificationModel;
import org.ejml.equation.Equation;
import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.Map;

public class LogisticRegressionClassifier extends ClassificationModel {

    public SimpleMatrix beta;

    @Override
    public void fit(Dataset dataset) {
        SimpleMatrix predictorsMatrix = new SimpleMatrix(dataset.getPredictors());
        SimpleMatrix ones = new SimpleMatrix(predictorsMatrix.getNumRows(), 1);
        ones.fill(1);
        SimpleMatrix designMatrix = ones.concatColumns(predictorsMatrix);
        beta = SimpleMatrix.random(designMatrix.getNumCols(), 1);
        var yTrue = new SimpleMatrix(dataset.getLabels());
        optimize(designMatrix, yTrue);
    }

    private void optimize(SimpleMatrix designMatrix, SimpleMatrix yTrue) {
        for (int i = 0; i < 10; i++) {
            SimpleMatrix z = designMatrix.mult(beta);
            SimpleMatrix prediction = z.elementExp().elementDiv(z.elementExp().plus(1)).elementLog();
            double loss = -yTrue.elementMult(prediction).elementSum();
            SimpleMatrix gradBeta = designMatrix.transpose().mult(prediction.minus(yTrue));
            beta = beta.minus(gradBeta.scale(0.01));
        }
    }

    @Override
    public Map<String, Double> evaluate(Dataset dataset) {
        return Map.of();
    }

    @Override
    public ArrayList<Integer> predict(ArrayList<ArrayList<Float>> data) {
        return null;
    }

    @Override
    public void log_odds() {

    }
}
