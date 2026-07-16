package pose.layer;

import pose.tensor.Tensor;

public class SigmoidLayer extends Layer
{
	@Override
	public Tensor forward(Tensor input)
	{
		this.input = input;
		
		output = Tensor.acquire(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());
		
		for(int i=0; i<input.size(); i++)
		{
			output.getData()[i] = 1f / (1f + (float)Math.exp(-input.getData()[i]));
		}
		//System.out.println("Sigmoid---min: " + output.min() + " | max: " + output.max() + " | mean: " + output.mean());
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor gradOutput = Tensor.acquire(output.getBatchSize(), output.getChannels(), output.getHeight(), output.getWidth());
		
		for(int i=0; i<output.size(); i++)
		{
			float sigmoid = output.getData()[i];
			
			gradOutput.getData()[i] = gradient.getData()[i] * sigmoid * (1f-sigmoid);
		}
		//System.out.println("sigmoid backward---min: " + gradOutput.min() + " | max: " + gradOutput.max() + " | mean: " + gradOutput.mean());
		return gradOutput;
	}
	
	public void updateWeights(float learningRate, float beta1, float beta2, float epsilon, int counter){}
}
