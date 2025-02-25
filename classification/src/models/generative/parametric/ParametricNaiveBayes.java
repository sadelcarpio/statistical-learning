package models.generative.parametric;

import org.ejml.simple.SimpleMatrix;

public class ParametricNaiveBayes extends ParametricGenerativeClassifier {

    public ParametricNaiveBayes(int nClasses) {
        super(nClasses);
    }

    @Override
    public void calculateCovMatrix(double[][] labels, double[][] predictors) {

    }

    @Override
    public double[][] predict(SimpleMatrix data) {
        return new double[0][];
    }
}
