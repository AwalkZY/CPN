def assert_param(param, field, field_type):
    assert field in param, "Param {} doesn't exist.".format(field)
    assert type(param[field]) is field_type, \
        'The type of {} is expected to be {}, but it\'s {} here instead.'.format(field, field_type, type(param[field]))


def assert_size(variable, variable_name, target_size):
    assert variable.size() == target_size, \
        'The shape of variable {} is expected to be {}, but it\'s {} here instead.'.format(variable_name, target_size,                                                                             variable.size())
