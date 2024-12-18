from beam import function


@function(cpu="100m", memory="100Mi")  # Each function runs on 100 millicores of CPU
def square(x):
    sum = 0

    for i in range(x):
        sum += i**2

    return {"sum": sum}


def main():
    print(square.local(x=10))
    print(square.remote(x=10))

    # Run a remote container for every item in list
    for i in square.map(range(5)):
        print(i)


if __name__ == "__main__":
    main()
