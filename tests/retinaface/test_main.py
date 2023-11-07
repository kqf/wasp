from wasp.retinaface.train import Paths, main


def test_main(tmp_dir):
    main(paths=Paths(tmp_dir, tmp_dir, tmp_dir, tmp_dir))
