package pose.layer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import pose.tensor.Tensor;

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

	private float[] weightM;
	private float[] weightV;
	private float[] biasM;
	private float[] biasV;

	private final static int NUM_THREADS = Runtime.getRuntime().availableProcessors();
	private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(NUM_THREADS);

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
		double limit = Math.sqrt(6.0 / fanIn);
		uniform(weights, limit);

		weightM = new float[weights.length];
		weightV = new float[weights.length];
		biasM = new float[bias.length];
		biasV = new float[bias.length];

		Arrays.fill(weightM, 0);
		Arrays.fill(weightV, 0);
		Arrays.fill(biasM, 0);
		Arrays.fill(biasV, 0);
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

		Tensor output = Tensor.acquire(batchSize, outChannels, outHeight, outWidth);

		float in[] = input.getData();
		float out[] = output.getData();

		int inHeightWidth = inHeight * inWidth;
		int outHeightWidth = outHeight * outWidth;
		int inChannelHeightWidth = inChannels * inHeightWidth;
		int outChannelHeightWidth = outChannels * outHeightWidth;

		int kSize = kernalSize;
		int kernalVolume = kSize * kSize;

		int channelsPerTask = Math.max(1, (outChannels + (NUM_THREADS * 2) - 1) / (NUM_THREADS * 2));

		List<Future<?>> futures = new ArrayList<Future<?>>();

		for (int startChannel = 0; startChannel < outChannels; startChannel += channelsPerTask)
		{
			int start = startChannel;
			int end = Math.min(start + channelsPerTask, outChannels);

			futures.add(EXECUTOR.submit(() -> {
				for (int batch = 0; batch < batchSize; batch++)
				{
					int inBatchOffset = batch * inChannelHeightWidth;
					int outBatchOffset = batch * outChannelHeightWidth;

					for (int channelOut = start; channelOut < end; channelOut++)
					{
						int outChannelOffset = channelOut * outHeightWidth;
						int weightOutChannelOffset = channelOut * (inChannels * kernalVolume);

						for (int outY = 0; outY < outHeight; outY++)
						{
							int inYBase = outY * stride - padding;
							int outIndex = outBatchOffset + outChannelOffset + outY * outWidth;

							for (int outX = 0; outX < outWidth; outX++, outIndex++)
							{
								int inXBase = outX * stride - padding;

								float sum = bias[channelOut];

								for (int channelIn = 0; channelIn < inChannels; channelIn++)
								{
									int inChannelOffset = inBatchOffset + channelIn * inHeightWidth;
									int weightInChannelOffset = weightOutChannelOffset + channelIn * kernalVolume;

									for (int kernalY = 0; kernalY < kernalSize; kernalY++)
									{
										int inY = inYBase + kernalY;

										if (inY < 0 || inY >= inHeight)
										{
											continue;
										}
										int inRowOffset = inChannelOffset + inY * inWidth;

										for (int kernalX = 0; kernalX < kernalSize; kernalX++)
										{
											int inX = inXBase + kernalX;

											if (inX < 0 || inX >= inWidth)
											{
												continue;
											}
											int inIndex = inRowOffset + inX;
											int weightIndex = weightInChannelOffset + kernalY * kSize + kernalX;

											sum += in[inIndex] * weights[weightIndex];
										}
									}
								}
								out[outIndex] = sum;
							}
						}
					}
				}
			}));
		}
		for (Future<?> future : futures)
		{
			try
			{
				future.get();
			}
			catch (InterruptedException e)
			{
				Thread.currentThread().interrupt();
				e.printStackTrace();
			}
			catch (ExecutionException e)
			{
				e.printStackTrace();
			}
		}
		// System.out.println("Conv forward---min: " + output.min() + " | max: " +
		// output.max() + " | mean: " + output.mean());

		return output;
	}

	@Override
	public Tensor backward(Tensor gradient)
	{
		Tensor gradientInput = Tensor.acquire(input.getBatchSize(), input.getChannels(), input.getHeight(), input.getWidth());

		Arrays.fill(weightGradients, 0f);
		Arrays.fill(biasGradients, 0f);

		float[] in = input.getData();
		float[] grad = gradient.getData();
		float[] gradIn = gradientInput.getData();

		int batchSize = input.getBatchSize();
		int inHeight = input.getHeight();
		int inWidth = input.getWidth();

		int outHeight = gradient.getHeight();
		int outWidth = gradient.getWidth();

		int inHeightWidth = inHeight * inWidth;
		int outHeightWidth = outHeight * outWidth;
		int inChannelHeightWidth = inChannels * inHeightWidth;
		int outChannelHeightWidth = outChannels * outHeightWidth;

		int kernalArea = kernalSize * kernalSize;
		int kernalVolume = kernalArea * inChannels;

		int threads = Math.min(outChannels, NUM_THREADS);

		List<Future<GradientAccumulator>> futures = new ArrayList<Future<GradientAccumulator>>(threads);

		for (int thread = 0; thread < threads; thread++)
		{
			int startChannel = thread * outChannels / threads;
			int endChannel = (thread + 1) * outChannels / threads;

			futures.add(EXECUTOR.submit(() -> {
				float[] localGradIn = new float[gradIn.length];
				float[] localWeightGradients = new float[weightGradients.length];
				float[] localBias = new float[bias.length];

				for (int batch = 0; batch < batchSize; batch++)
				{
					int inBatchOffset = batch * inChannelHeightWidth;
					int outBatchOffset = batch * outChannelHeightWidth;

					for (int channelOut = startChannel; channelOut < endChannel; channelOut++)
					{
						int gradChannelOffset = outBatchOffset + channelOut * outHeightWidth;
						int weightOutChannelOffset = channelOut * kernalVolume;

						float biasGrad = 0f;

						for (int outY = 0; outY < outHeight; outY++)
						{
							int gradIndex = gradChannelOffset + outY * outWidth;
							int inYBase = outY * stride - padding;

							for (int outX = 0; outX < outWidth; outX++, gradIndex++)
							{
								float g = grad[gradIndex];
								biasGrad += g;

								int inXBase = outX * stride - padding;

								for (int channelIn = 0; channelIn < inChannels; channelIn++)
								{
									int inputChannelOffset = inBatchOffset + channelIn * inHeightWidth;
									int weightInChannelOffset = weightOutChannelOffset + channelIn * kernalArea;
									int weightIndex = weightInChannelOffset;

									for (int kernalY = 0; kernalY < kernalSize; kernalY++)
									{
										int inY = inYBase + kernalY;

										if (inY < 0 || inY >= inHeight)
										{
											weightIndex += kernalSize;
											continue;
										}
										int inputRow = inputChannelOffset + inY * inWidth;

										for (int kernalX = 0; kernalX < kernalSize; kernalX++, weightIndex++)
										{
											int inX = inXBase + kernalX;

											if (inX < 0 || inX >= inWidth)
											{
												continue;
											}
											int inputIndex = inputRow + inX;

											localWeightGradients[weightIndex] += g * in[inputIndex];
											localGradIn[inputIndex] += g * weights[weightIndex];
										}
									}
								}
							}
						}
						localBias[channelOut] += biasGrad;
					}
				}
				return new GradientAccumulator(localGradIn, localWeightGradients, localBias);
			}));
		}
		try
		{
			for (Future<GradientAccumulator> future : futures)
			{
				GradientAccumulator accumulator = future.get();

				float[] localGradIn = accumulator.getGradientInput();
				float[] localWeightGradient = accumulator.getWeightGradient();
				float[] localBias = accumulator.getBias();

				for (int i = 0; i < gradIn.length; i++)
				{
					gradIn[i] += localGradIn[i];
				}
				for (int i = 0; i < weightGradients.length; i++)
				{
					weightGradients[i] += localWeightGradient[i];
				}
				for (int i = 0; i < biasGradients.length; i++)
				{
					biasGradients[i] += localBias[i];
				}
			}
		}
		catch (InterruptedException e)
		{
			Thread.currentThread().interrupt();
			e.printStackTrace();
		}
		catch (ExecutionException e)
		{
			e.printStackTrace();
		}
		// System.out.println("Conv backward---min: " + gradientInput.min() + " | max: "
		// + gradientInput.max() + " | mean: " + gradientInput.mean());

		return gradientInput;
	}

	@Override
	public void updateWeights(float learningRate, float beta1, float beta2, float epsilon, int counter)
	{
		for (int i = 0; i < weights.length; i++)
		{
			weightM[i] = beta1 * weightM[i] + (1 - beta1) * weightGradients[i];
			weightV[i] = beta2 * weightV[i] + (1 - beta2) * weightGradients[i] * weightGradients[i];

			float mHat = weightM[i] / (1 - (float) Math.pow(beta1, counter));
			float vHat = weightV[i] / (1 - (float) Math.pow(beta2, counter));

			weights[i] -= learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon);
		}
		for (int i = 0; i < bias.length; i++)
		{
			biasM[i] = beta1 * biasM[i] + (1 - beta1) * biasGradients[i];
			biasV[i] = beta2 * biasV[i] + (1 - beta2) * biasGradients[i] * biasGradients[i];

			float mHat = biasM[i] / (1 - (float) Math.pow(beta1, counter));
			float vHat = biasV[i] / (1 - (float) Math.pow(beta2, counter));

			bias[i] -= learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon);
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
