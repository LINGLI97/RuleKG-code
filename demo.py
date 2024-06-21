OLD = 'http://oldboyedu.com'


def bb(OLD_URL):
    #global OLD

    OLD = OLD_URL + OLD
    print("内部"+OLD_URL)


if __name__ == '__main__':
    bb(OLD)
    print("外部"+OLD)