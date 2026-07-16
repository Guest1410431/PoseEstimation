package pose.loss;

import pose.tensor.Tensor;

public class MSELoss implements LossFunction
{
	private static final float POS_WEIGHT = 100f;

	@Override
	public float forward(Tensor prediction, Tensor target)
	{
		float loss = 0f;

		for (int i = 0; i < prediction.size(); i++)
		{
			float diff = prediction.getData()[i] - target.getData()[i];
			float weight = target.getData()[i] > 0f ? POS_WEIGHT : 1f;

			loss += weight * diff * diff;
		}
		return loss / prediction.size();
	}

	@Override
	public Tensor backward(Tensor prediction, Tensor target)
	{
		Tensor gradient = Tensor.acquire(prediction.getBatchSize(), prediction.getChannels(), prediction.getHeight(), prediction.getWidth());

		for (int i = 0; i < prediction.size(); i++)
		{
			float weight = target.getData()[i] > 0f ? POS_WEIGHT : 1f;
			gradient.getData()[i] = weight * 2f * (prediction.getData()[i] - target.getData()[i]) / prediction.size();
		}
		return gradient;
	}
}
