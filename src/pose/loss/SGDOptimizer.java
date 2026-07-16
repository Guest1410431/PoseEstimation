package pose.loss;

import pose.layer.Layer;

public class SGDOptimizer implements Optimizer
{
	private float learningRate;
	private float beta1;
	private float beta2;
	private float epsilon;
	private int counter;

	public SGDOptimizer(float learningRate, float beta1, float beta2, float epsilon, int counter)
	{
		this.learningRate = learningRate;
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.epsilon = epsilon;
		
		this.counter = counter;
	}

	@Override
	public void step(Layer layer)
	{
		layer.updateWeights(learningRate, beta1, beta2, epsilon, counter);

		counter++;
	}
}
