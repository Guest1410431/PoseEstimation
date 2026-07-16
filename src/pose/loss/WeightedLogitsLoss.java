package pose.loss;

import pose.tensor.Tensor;

public class WeightedLogitsLoss implements LossFunction
{
	private static final float POS_WEIGHT = 100f;

	@Override
	public float forward(Tensor prediction, Tensor target)
	{
		float loss = 0f;

		for (int i = 0; i < prediction.size(); i++)
		{
			float z = prediction.getData()[i];
			float t = target.getData()[i];
			float weight = t > 0f ? POS_WEIGHT : 1f;

			float stable = Math.max(z, 0f) - z * t + (float) Math.log(1f + Math.exp(-Math.abs(z)));

			loss += weight * stable;
		}
		return loss / prediction.size();
	}

	@Override
	public Tensor backward(Tensor prediction, Tensor target)
	{
		Tensor gradient = Tensor.acquire(prediction.getBatchSize(), prediction.getChannels(), prediction.getHeight(), prediction.getWidth());

		for (int i = 0; i < prediction.size(); i++)
		{
			float z = prediction.getData()[i];
			float t = target.getData()[i];
			float weight = t > 0f ? POS_WEIGHT : 1f;

			float sigmoid = 1f / (1f + (float) Math.exp(-z));

			gradient.getData()[i] = weight * (sigmoid - t) / prediction.size();
		}
		return gradient;
	}

}
