import libfiltre as lib


def mainfaction():
    list_fram = lib.open_file("simple440add1000.wav")
    list_fram = lib.channels2one( list_fram )
    #list_fram = lib.filtrephaut2( list_fram, 440)
    lib.writefile(list_fram,"out-test.wav")

mainfaction()