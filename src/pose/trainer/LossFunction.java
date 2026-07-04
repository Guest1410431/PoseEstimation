package pose.trainer;

import pose.tensor.Tensor;

public interface LossFunction
{
	float forward(Tensor prediction, Tensor target);

	Tensor backward(Tensor prediction, Tensor target);
}
