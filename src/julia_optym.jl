module julia_optym

using LinearAlgebra

"Parameter used in SVGoldenSection method"
goldenRatio = (1 + sqrt(5)) / 2

"Modified golden section search method from previous task"
function goldenSearch(f, values, d, ϵ, maxiter)
  a, b = [-10, 10]  # Learning rate from [-10,10] is enough (based on the available informations)
  i = 0

  while abs(a - b) >= ϵ && i <= maxiter
    i += 1
    # Calculate next 2 points
    x1 = b - (b - a) / goldenRatio
    x2 = a + (b - a) / goldenRatio
    fArgs1 = broadcast(-, values, x1 * d)
    fArgs2 = broadcast(-, values, x2 * d)
    # Compare function values from calculated points and determine where minimum is located
    if f(fArgs1...) < f(fArgs2...)
      b = x2
    else
      a = x1
    end
  end
  # Get the approximate value and return it
  result = (a + b) / 2
  return result
end

"Function taken from algorithm"
function derivate(f, values, ϵ=1e-6)
  result = []
  n = length(values)

  for i = 1:n
    zer = zeros(n)
    zer[i] = ϵ
    d_values = broadcast(+, values, zer)
    append!(result, (f(d_values...) - f(values...)) / ϵ)
  end

  return result
end

function difference(values1, values2)
  n = length(values1)
  value = 0

  for i = 1:n
    value += abs(values1[i] - values2[i])
  end

  result = value / n
  return result
end

function steepestDescent(f, starting_values, ϵ=1e-6, maxiter=200)
  values, prev_values = starting_values, starting_values

  while true
    d = derivate(f, values)
    learning_rate = goldenSearch(f, values, d, ϵ, maxiter)
    prev_values = values
    values = broadcast(-, values, learning_rate * d)

    if difference(values, prev_values) < ϵ
      return values
    end
  end
end

export steepestDescent

end # module