from lmas.io.project import PROJECT_FORMAT
from lmas.profiles import PROFILE_FORMAT, startup_profile


def test_v1_formats_and_writer_version():
    assert PROJECT_FORMAT == 'lmas-project-v1.1'
    assert PROFILE_FORMAT == 'lmas-profile-v1.0'
    assert startup_profile().to_dict()['lmas_version'] == '1.6.2'
