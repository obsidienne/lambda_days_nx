<<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
  File.read!("tmp/train-images-idx3-ubyte.gz") |> :zlib.gunzip()

heatmap =
  images
  |> Nx.from_binary({:u, 8})
  |> Nx.reshape({n_images, n_rows, n_cols})
  |> Nx.to_heatmap()

images =
  images
  |> Nx.from_binary({:u, 8})
  |> Nx.reshape({n_images, n_rows * n_cols}, names: [:batch, :input])
  |> Nx.divide(255)
  |> Nx.to_batched_list(30)

<<_::32, n_labels::32, labels::binary>> =
  File.read!("tmp/train-labels-idx1-ubyte.gz") |> :zlib.gunzip()

labels =
  labels
  |> Nx.from_binary({:u, 8})
  |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
  |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
  |> Nx.to_batched_list(30)

defmodule MNIST do
  import Nx.Defn

  @default_defn_compiler EXLA

  defn init_params do
    w1 = Nx.random_normal({784, 128}, 0.0, 0.1, names: [:input, :hidden])
    b1 = Nx.random_normal({128}, 0.0, 0.1, names: [:hidden])
    w2 = Nx.random_normal({128, 10}, 0.0, 0.1, names: [:hidden, :output])
    b2 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])
    {w1, b1, w2, b2}
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [:output], keep_axes: true)
  end

  defn loss({w1, b1, w2, b2}, images, labels) do
    preds = predict({w1, b1, w2, b2}, images)
    -Nx.sum(Nx.mean(Nx.log(preds) * labels, axes: [:output]))
  end

  defn update({w1, b1, w2, b2} = params, images, labels) do
  {grad_w1, grad_b1, grad_w2, grad_b2} =
    grad(params, loss(params, images, labels))

    {w1 - grad_w1 * 0.01, b1 - grad_b1 * 0.01, w2 - grad_w2 * 0.01, b2 - grad_b2 * 0.01 }
  end

end

zip = Enum.zip(images, labels) |> Enum.with_index()

params =
  for e <- 1..5,
    {{images, labels}, b} <- zip,
    reduce: MNIST.init_params() do
    params ->
      IO.puts "epoch #{e}, batch #{b}"
      MNIST.update(params, images, labels)
    end
