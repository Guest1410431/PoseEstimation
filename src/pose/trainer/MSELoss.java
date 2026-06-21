package pose.trainer;

import pose.layer.Tensor;

public class MSELoss implements LossFunction
{
	@Override
	public float forward(Tensor prediction, Tensor target)
	{
		float loss = 0f;
		
		for(int i=0; i<prediction.size(); i++)
		{
			float diff = prediction.getData()[i] - target.getData()[i];
			loss += diff * diff;
		}
		return loss / prediction.size();
	}

	@Override
	public Tensor backward(Tensor prediction, Tensor target)
	{	
		Tensor gradient = new Tensor(prediction.getBatchSize(), prediction.getChannels(), prediction.getHeight(), prediction.getWidth());
		
		for(int i=0; i<prediction.size(); i++)
		{
			gradient.getData()[i] = 2f * (prediction.getData()[i] - target.getData()[i]) / prediction.size();
		}
		return gradient;
	}

}
