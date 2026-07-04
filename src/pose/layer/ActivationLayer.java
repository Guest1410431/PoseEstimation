package pose.layer;

import pose.tensor.Tensor;

public class ActivationLayer extends Layer
{
	@Override
	public Tensor forward(Tensor input)
	{
		this.input = input;
		
		output = Tensor.acquire(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());
		
		for(int i=0; i<input.size(); i++)
		{
			output.getData()[i] = Math.max(0f, input.getData()[i]);
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor gradientInput = Tensor.acquire(output.getBatchSize(), output.getChannels(), output.getHeight(), output.getWidth());
		
		for(int i=0; i<output.size(); i++)
		{
			gradientInput.getData()[i] = output.getData()[i] > 0 ? gradient.getData()[i] : 0;
		}
		return gradientInput;
	}
}
