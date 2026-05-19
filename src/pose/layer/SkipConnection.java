package pose.layer;

public class SkipConnection
{
	private Tensor encoderOutput;

	public void save(Tensor tensor)
	{
		this.encoderOutput = tensor;
	}

	public Tensor getEncoderOutput()
	{
		return encoderOutput;
	}
}
