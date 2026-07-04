package pose.layer;

import pose.tensor.Tensor;

public class DownSampleLayer extends Layer
{
	private final int kernalSize;
	private final int stride;

	private int[] maxIndices;

	public DownSampleLayer(int kernalSize, int stride)
	{
		this.kernalSize = kernalSize;
		this.stride = stride;
	}

	@Override
	public Tensor forward(Tensor input)
	{
		// Max pooling
		this.input = input;

		int batchSize = input.getBatchSize();
		int channels = input.getChannels();

		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = (inHeight - kernalSize) / stride + 1;
		int outWidth = (inWidth - kernalSize) / stride + 1;

		Tensor output = Tensor.acquire(batchSize, channels, outHeight, outWidth);

		float[] in = input.getData();
		float[] out = output.getData();

		int inHeightWidth = inHeight * inWidth;
		int outHeightWidth = outHeight * outWidth;

		int outChannelHeightWidth = channels * outHeightWidth;

		maxIndices = new int[batchSize * outChannelHeightWidth];

		for (int batch = 0; batch < batchSize; batch++)
		{
			int inBatch = batch * channels * inHeightWidth;
			int outBatch = batch * channels * outHeightWidth;

			for (int channel = 0; channel < channels; channel++)
			{
				int inChannel = inBatch + channel * inHeightWidth;
				int outChannel = outBatch + channel * outHeightWidth;

				int outIndex = outChannel;

				for (int outY = 0; outY < outHeight; outY++)
				{
					int inYBase = outY * stride;

					for (int outX = 0; outX < outWidth; outX++, outChannel++)
					{
						int inXBase = outX * stride;

						float max = -Float.MAX_VALUE;
						int maxIndex = -1;

						for (int kernalY = 0; kernalY < kernalSize; kernalY++)
						{
							int row = inChannel + (inYBase + kernalY) * inWidth + inXBase;

							for (int kernalX = 0; kernalX < kernalSize; kernalX++)
							{
								int idx = row + kernalX;
								float value = in[idx];

								if (value > max)
								{
									max = value;
									maxIndex = idx;
								}
							}
						}
						out[outIndex] = max;
	                    maxIndices[outIndex] = maxIndex;
					}
				}
			}
		}
		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor gradientInput = Tensor.acquire(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());

		float[] grad = gradient.getData();
		float[] gradIn = gradientInput.getData();

		for (int i = 0; i < grad.length; i++)
		{
			gradIn[maxIndices[i]] += grad[i];
		}
		return gradientInput;
	}
}
