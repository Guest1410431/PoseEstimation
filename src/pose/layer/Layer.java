package pose.layer;

import pose.tensor.Tensor;

public abstract class Layer
{
	protected Tensor input;
	protected Tensor output;

	public abstract Tensor forward(Tensor input);

	public abstract Tensor backward(Tensor gradient);

	public void updateWeights(float learningRate){};
}
