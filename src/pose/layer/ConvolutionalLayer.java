package pose.layer;

import java.util.Arrays;
import java.util.Random;

public class ConvolutionalLayer extends Layer
{
	private final int inChannels;
	private final int outChannels;
	private final int kernalSize;
	private final int stride;
	private final int padding;
	private int batchSize;

	private float[] weights;
	private final float[] bias;
	private float[] weightGradients;
	private float[] biasGradients;

	public ConvolutionalLayer(int inChannels, int outChannels, int kernalSize, int stride, int padding)
	{
		this.inChannels = inChannels;
		this.outChannels = outChannels;
		this.kernalSize = kernalSize;
		this.stride = stride;
		this.padding = padding;

		int weightCount = outChannels * inChannels * kernalSize * kernalSize;
		this.weights = new float[weightCount];
		this.bias = new float[outChannels];
		this.weightGradients = new float[weights.length];
		this.biasGradients = new float[bias.length];

		// He/Kaiming Weight Distribution
		int fanIn = inChannels * kernalSize * kernalSize;
		double limit = Math.sqrt(6 / fanIn) * Math.sqrt(2);
		uniform(weights, limit);
	}

	@Override
	public Tensor forward(Tensor input)
	{
		this.input = input;

		batchSize = input.getBatchSize();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = Math.floorDiv(inHeight + (padding * 2) - kernalSize, stride) + 1;
		int outWidth = Math.floorDiv(inWidth + (padding * 2) - kernalSize, stride) + 1;

		Tensor output = new Tensor(batchSize, outChannels, outHeight, outWidth);

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int chout = 0; chout < outChannels; chout++)
			{
				for (int outY = 0; outY < outHeight; outY++)
				{
					for (int outX = 0; outX < outWidth; outX++)
					{
						float sum = bias[chout];

						for (int chin = 0; chin < inChannels; chin++)
						{
							for (int kernalY = 0; kernalY < kernalSize; kernalY++)
							{
								for (int kernalX = 0; kernalX < kernalSize; kernalX++)
								{
									int inY = outY * stride + kernalY - padding;
									int inX = outX * stride + kernalX - padding;

									if (inY < 0 || inX < 0 || inY >= inHeight || inX >= inWidth)
									{
										continue;
									}
									int weightIndex = chout * (kernalSize * kernalSize * inChannels) + kernalY * (kernalSize * inChannels) + kernalX * inChannels + chin;

									sum += input.get(batch, chin, inY, inX) * weights[weightIndex];
								}
							}
						}
						output.set(batch, chout, outY, outX, sum);
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

		Arrays.fill(weightGradients, 0f);
		Arrays.fill(biasGradients, 0f);

		int batchSize = input.getBatchSize();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = gradient.getHeight();
		int outWidth = gradient.getWidth();

		float[] inputData = input.getData();
		float[] gradData = gradient.getData();
		float[] gradInputData = gradientInput.getData();

		for (int batch = 0; batch < batchSize; batch++)
		{
			for (int outY = 0; outY < outHeight; outY++)
			{
				for (int outX = 0; outX < outWidth; outX++)
				{
					for (int chout = 0; chout < outChannels; chout++)
					{
						float grad = gradient.get(batch, chout, outY, outX);

						biasGradients[chout] += grad;

						for (int kernalY = 0; kernalY < kernalSize; kernalY++)
						{
							for (int kernalX = 0; kernalX < kernalSize; kernalX++)
							{
								int inY = outY * stride + kernalY - padding;
								int inX = outX * stride + kernalX - padding;

								if (inY < 0 || inX < 0 || inY >= inHeight || inX >= inWidth)
								{
									continue;
								}
								for (int chin = 0; chin < inChannels; chin++)
								{
									float inputValue = input.get(batch, chin, inY, inX);

									int weightIndex = chout * (kernalSize * kernalSize * inChannels) + kernalY * (kernalSize * inChannels) + kernalX * inChannels + chin;

									weightGradients[weightIndex] += grad * inputValue;

									gradientInput.add(batch, chin, inY, inX, grad * weights[weightIndex]);
								}
							}
						}
					}
				}
			}
		}
		return gradientInput;
	}

	@Override
	public void updateWeights(float learningRate)
	{
		float scale = 1.0f / batchSize;

		for (int i = 0; i < weights.length; i++)
		{
			weights[i] -= learningRate * weightGradients[i] * scale;
		}
		for (int i = 0; i < bias.length; i++)
		{
			bias[i] -= learningRate * biasGradients[i] * scale;
		}
	}

	private static void uniform(float[] array, double limit)
	{
		Random random = new Random();

		for (int i = 0; i < array.length; i++)
		{
			array[i] = (float) ((random.nextDouble() * 2 - 1) * limit);
		}
	}

	public float[] getWeights()
	{
		return weights;
	}

	public void setWeights(float[] weights)
	{
		this.weights = weights;
	}

	public float[] getWeightGradients()
	{
		return weightGradients;
	}
}
