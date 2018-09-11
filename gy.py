import libFfiltre as lib


def mainfaction():
    wavFilter  = lib.wavFilter()
    list_fram = wavFilter.open_file("simple440add1000.wav")
    list_fram = wavFilter.channels2one( list_fram )
    
    arr=[[128,0.1],[500,1]]
    wavFilter.CaculateTranLossPrecent(arr)

    list_fram  = wavFilter.fourrier( list_fram )
    #list_fram = wavFilter.filtrephaut2( list_fram, 440)

    wavFilter.writefile(list_fram,"out-test.wav")

mainfaction()