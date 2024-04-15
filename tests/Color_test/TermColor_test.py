from Blu.Utils.TermColor import printC, paintStr, setColor, resetColor

def test_1TermColor():
    print()
    print("TermColor test 1:")
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
    resetColor()
    print("The terminal color is now reset")
    print("test")
    print("test")
    print("TermColor Test 1 Complete")
