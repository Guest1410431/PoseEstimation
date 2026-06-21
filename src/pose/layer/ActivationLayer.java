package pose.layer;

public class ActivationLayer extends Layer
{
	@Override
	public Tensor forward(Tensor input)
	{
		float[] data = input.getData();

		for (int i = 0; i < data.length; i++)
		{
			if (data[i] < 0)
			{
				data[i] = 0;
			}
		}
		return input;
	}

	@Override
	public Tensor backward(Tensor input)
	{
		// TODO Auto-generated method stub
		return null;
	}
}
