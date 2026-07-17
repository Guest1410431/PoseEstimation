package pose.layer;

import pose.tensor.Tensor;

public class SkipConnection
{
	private Tensor encoderOutput;
	private Tensor gradient;

	public void save(Tensor tensor)
	{
		this.encoderOutput = tensor;
	}

	public Tensor getEncoderOutput()
	{
		return encoderOutput;
	}

	public Tensor getGradient()
	{
		return gradient;
	}

	public void setGradient(Tensor gradient)
	{
		this.gradient = gradient;
	}
	public void clear()
	{
		encoderOutput = null;
		gradient = null;
	}

	public void releaseEncoderOutput()
	{
		if(encoderOutput != null)
		{
			Tensor.release(encoderOutput);
			encoderOutput = null;
		}
	}
	public void releaseGradient()
	{
		if(gradient != null)
		{
			Tensor.release(gradient);
			gradient = null;
		}
	}
}
