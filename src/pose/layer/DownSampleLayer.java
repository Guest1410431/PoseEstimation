package pose.layer;

public class DownSampleLayer extends Layer
{
	private final int kernalSize;
	private final int stride;

	private int[][][][] maxValIndexY;
	private int[][][][] maxValIndexX;

	public DownSampleLayer(int kernalSize, int stride)
	{
		this.kernalSize = kernalSize;
		this.stride = stride;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// Max pooling

		int batchSize = input.getBatchSize();
		int channels = input.getChannels();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = (inHeight - kernalSize) / stride + 1;
		int outWidth = (inWidth - kernalSize) / stride + 1;

		Tensor output = new Tensor(batchSize, channels, outHeight, outWidth);

		maxValIndexY = new int[batchSize][channels][outHeight][outWidth];
		maxValIndexX = new int[batchSize][channels][outHeight][outWidth];
		
		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int channel = 0; channel < channels; channel++)
			{
				for (int outY = 0; outY < outHeight; outY++)
				{
					for (int outX = 0; outX < outWidth; outX++)
					{
						float maxVal = -Float.MAX_VALUE;
						int maxIndexX = -1;
						int maxIndexY = -1;

						for (int y = 0; y < kernalSize; y++)
						{
							for (int x = 0; x < kernalSize; x++)
							{
								int inY = outY * stride + y;
								int inX = outX * stride + x;

								float value = input.get(batch, channel, inY, inX);

								if (value > maxVal)
								{
									maxVal = value;

									maxIndexY = inY;
									maxIndexX = inX;
								}
							}
						}
						output.set(batch, channel, outY, outX, maxVal);
						maxValIndexY[batch][channel][outY][outX] = maxIndexY;
						maxValIndexX[batch][channel][outY][outX] = maxIndexX;
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor gradientInput = new Tensor(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());

		for (int batch = 0; batch < gradient.getBatchSize(); batch++)
		{
			for (int channel = 0; channel < gradient.getChannels(); channel++)
			{
				for (int outY = 0; outY < gradient.getHeight(); outY++)
				{
					for (int outX = 0; outX < gradient.getWidth(); outX++)
					{
						float grad = gradient.get(batch, channel, outY, outX);
						int maxX = maxValIndexX[batch][channel][outY][outX];
						int maxY = maxValIndexY[batch][channel][outY][outX];

						gradientInput.add(batch, channel, maxY, maxX, grad);
					}
				}
			}
		}
		return gradientInput;
	}
}
