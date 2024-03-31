from Blu.Utils.TermColor import printC, paintStr, setColor, resetColor

print("\n--TermColor Test Battery--\n")


def test_1TermColor():
    print()
    printC(text="This text is tea green",
           textColor="tea green")
    print(paintStr(text="This test has a seafoam green foreground and blood orange background",
                   textColor="seafoam green",
                   bgColor="blood orange"))

    setColor("red",
             "blue")
    print("This regular print statement is colored")
    print("test")
    print("test")
    print("test")
    resetColor()
    print("The terminal color is now reset")
    print("test")
    print("test")
    print("test")


print("\n--End TermColor Test Battery--\n")
