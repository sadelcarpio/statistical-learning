package models.generative.nonparametric;

import org.ejml.simple.SimpleMatrix;

public class NonParametricNaiveBayes extends NonParametricGenerativeClassifier {

    @Override
    public double[][] predict(SimpleMatrix data) {
        return new double[0][];
    }

    @Override
    public void logOdds() {

    }
}
