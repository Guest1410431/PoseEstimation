package pose.layer;

import pose.tensor.Tensor;

public abstract class Layer
{
	protected Tensor input;
	protected Tensor output;

	public abstract Tensor forward(Tensor input);

	public abstract Tensor backward(Tensor gradient);

	public abstract void updateWeights(float learningRate, float beta1, float beta2, float epsilon, int counter);
}
