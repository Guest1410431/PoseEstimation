package pose.trainer;

import pose.encoder.Network;
import pose.layer.Layer;

public class SGDOptimizer implements Optimizer
{
	private float learningRate;

	public SGDOptimizer(float learningRate)
	{
		this.learningRate = learningRate;
	}

	@Override
	public void step(Network network)
	{
		for (Layer layer : network.getLayers())
		{
			layer.updateWeights(learningRate);
		}
	}
}
