# LambdaDaysNx

**TODO: Add description**

## Installation

If [available in Hex](https://hex.pm/docs/publish), the package can be installed
by adding `lambda_days_nx` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:lambda_days_nx, "~> 0.1.0"}
  ]
end
```

Documentation can be generated with [ExDoc](https://github.com/elixir-lang/ex_doc)
and published on [HexDocs](https://hexdocs.pm). Once published, the docs can
be found at [https://hexdocs.pm/lambda_days_nx](https://hexdocs.pm/lambda_days_nx).

Download MNIST Database from the web site http://yann.lecun.com/exdb/mnist/
- train-images.idx3-ubyte.gz
- train-labels.idx1-ubyte.gz


```
iex(5)> b = File.read!("tmp/train-images-idx3-ubyte.gz") |> :zlib.gunzip()
<<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ...>>
iex(6)> <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> = b
<<0, 0, 8, 3, 0, 0, 234, 96, 0, 0, 0, 28, 0, 0, 0, 28, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  ...>>
iex(7)> n_images
60000
iex(8)> n_rows
28
iex(9)> n_cols
28
iex(10)> t = Nx.from_binary(images, {:u, 8})
#Nx.Tensor<
  u8[47040000]
  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...]
>
iex(12)> t = Nx.reshape(t, {n_images, n_rows, n_cols})
#Nx.Tensor<
  u8[60000][28][28]
  [
    [
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
      ...
    ],
    ...
  ]
>
iex(13)> t = Nx.reshape(t, {n_images, n_rows * n_cols})
#Nx.Tensor<
  u8[60000][784]
  [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...],
    ...
  ]
>
iex(6)> t = Nx.divide(t, 255)
#Nx.Tensor<
  f32[60000][784]
  [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
    ...
  ]
>
iex(7)> Nx.to_batched_list(t, 30)
[#Nx.Tensor<
    f32[30][784]
    [
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...],
      ...

iex(10)> l = File.read!("tmp/train-labels-idx1-ubyte.gz") |> :zlib.gunzip()
<<0, 0, 8, 1, 0, 0, 234, 96, 5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2,
  8, 6, 9, 4, 0, 9, 1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8,
  ...>>
iex(11)> <<_::32, n_labels::32, labels::binary>> = l
<<0, 0, 8, 1, 0, 0, 234, 96, 5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2,
  8, 6, 9, 4, 0, 9, 1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8,
  ...>>
iex(12)> n_labels
60000
iex(13)> labels
<<5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9, 1, 1, 2,
  4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9, 8, 5, 9, 3, ...>>
iex(14)> t = Nx.from_binary(labels, {:u, 8})
#Nx.Tensor<
  u8[60000]
  [5, 0, 4, 1, 9, 2, 1, 3, 1, 4, 3, 5, 3, 6, 1, 7, 2, 8, 6, 9, 4, 0, 9, 1, 1, 2, 4, 3, 2, 7, 3, 8, 6, 9, 0, 5, 6, 0, 7, 6, 1, 8, 7, 9, 3, 9, 8, 5, 9, 3, ...]
> 
iex(16)> t = Nx.reshape(t, {60_000, 1})
    [3],
    ...
  ] 
>
iex(17)> o = Nx.tensor(Enum.to_list(0..9))
#Nx.Tensor<
  s64[10]
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
>
iex(18)> t = Nx.equal(t, o)
#Nx.Tensor<
  u8[60000][10]
  [
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ...
  ]
>
iex(19)> Nx.to_batched_list(t, 30)
```


