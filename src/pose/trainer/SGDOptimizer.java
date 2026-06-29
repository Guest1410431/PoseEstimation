package pose.trainer;

import pose.layer.Layer;

public class SGDOptimizer implements Optimizer
{
	private float learningRate;

	public SGDOptimizer(float learningRate)
	{
		this.learningRate = learningRate;
	}

	@Override
	public void step(Layer layer)
	{
		layer.updateWeights(learningRate);
	}
}
