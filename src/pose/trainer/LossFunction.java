package pose.trainer;

import pose.layer.Tensor;

public interface LossFunction
{
	float forward(Tensor prediction, Tensor target);

	Tensor backward(Tensor prediction, Tensor target);
}
