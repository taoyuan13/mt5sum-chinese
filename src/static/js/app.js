var data = []
var token = ""

jQuery(document).ready(function () {
    var slider = $('#max_words')
    slider.on('change mousemove', function (evt) {
        $('#label_max_words').text('最大长度: ' + slider.val())
    })

    var slider2 = $('#num_beams')
    slider2.on('change mousemove', function (evt) {
        $('#label_num_beams').text('搜索束宽: ' + slider2.val())
    })

    var slider3 = $('#no_repeat')
    slider3.on('change mousemove', function (evt) {
        $('#label_no_repeat').text('不重复元: ' + slider3.val())
    })

    $('#btn-process').on('click', function () {
        input_text = $('#txt_input').val()
        // model = $('#input_model').val()
        num_words = $('#max_words').val()
        num_beams = $('#num_beams').val()
        no_repeat = $('#no_repeat').val()
        $.ajax({
            url: '/predict',
            type: "post",
            contentType: "application/json",
            dataType: "json",
            data: JSON.stringify({
                "input_text": input_text,
                "num_words": num_words,
                "num_beams": num_beams,
                "no_repeat": no_repeat,
            }),
            beforeSend: function () {
                $('.overlay').show()
            },
            complete: function () {
                $('.overlay').hide()
            }
        }).done(function (jsondata, textStatus, jqXHR) {
            console.log(jsondata)
            $('#input_summary').val(jsondata['response']['summary'])
        }).fail(function (jsondata, textStatus, jqXHR) {
            alert(jsondata['responseJSON']['message'])
        });
    })


})