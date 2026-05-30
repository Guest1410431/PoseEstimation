package pose.layer;

public class SigmoidLayer extends Layer
{
	@Override
	public Tensor forward(Tensor input)
	{
		float[] data = input.getData();

		for (int i = 0; i < data.length; i++)
		{
			data[i] = 1f / (1f + (float)Math.exp(-data[i]));
		}
		return input;
	}
}
