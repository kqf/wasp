from wasp.retinaface.train import Paths, main


def test_main(tmp_path):
    main(paths=Paths(tmp_path, tmp_path, tmp_path, tmp_path))
