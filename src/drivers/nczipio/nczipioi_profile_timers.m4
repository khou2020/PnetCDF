define(`TIMERS', `( `(`total', `0')', dnl
                    `(`init', `1')', dnl
                    `(`init_meta', `2')', dnl
                    `(`init_csize', `3')', dnl
                    `(`init_cown', `4')', dnl
                    `(`resize', `5')', dnl
                    `(`put', `6')', dnl
                    `(`put_cb', `7')', dnl
                    `(`put_cb_init', `8')', dnl   
                    `(`put_cb_sync', `9')', dnl   
                    `(`put_cb_pack_req', `10')', dnl   
                    `(`put_cb_pack_rep', `11')', dnl 
                    `(`put_cb_unpack_req', `12')', dnl 
                    `(`put_cb_unpack_rep', `13')', dnl 
                    `(`put_cb_send_req', `14')', dnl 
                    `(`put_cb_send_rep', `15')', dnl  
                    `(`put_cb_recv_req', `16')', dnl  
                    `(`put_cb_recv_rep', `17')', dnl  
                    `(`put_cb_self', `18')', dnl   
                    `(`put_io', `19')', dnl
                    `(`put_io_wr', `20')', dnl
                    `(`put_io_rd', `21')', dnl
                    `(`put_io_com', `22')', dnl
                    `(`put_io_decom', `23')', dnl
                    `(`put_io_sync', `24')', dnl
                    `(`put_io_init', `25')', dnl
                    `(`get', `26')', dnl
                    `(`get_resize', `27')', dnl
                    `(`get_cb', `28')', dnl
                    `(`get_cb_init', `29')', dnl   
                    `(`get_cb_sync', `30')', dnl   
                    `(`get_cb_pack_req', `31')', dnl   
                    `(`get_cb_pack_rep', `32')', dnl 
                    `(`get_cb_unpack_req', `33')', dnl 
                    `(`get_cb_unpack_rep', `34')', dnl 
                    `(`get_cb_send_req', `35')', dnl 
                    `(`get_cb_send_rep', `36')', dnl  
                    `(`get_cb_recv_req', `37')', dnl  
                    `(`get_cb_recv_rep', `38')', dnl  
                    `(`get_cb_self', `39')', dnl   
                    `(`get_io', `40')', dnl
                    `(`get_io_wr', `41')', dnl
                    `(`get_io_rd', `42')', dnl
                    `(`get_io_com', `43')', dnl
                    `(`get_io_decom', `44')', dnl
                    `(`get_io_sync', `45')', dnl
                    `(`get_io_init', `46')', dnl
                    `(`finalize', `47')', dnl
                    `(`finalize_meta', `48')', dnl
                    `(`nb', `49')', dnl
                    `(`nb_post', `50')', dnl
                    `(`nb_wait', `51')', dnl
)')dnl
define(`NTIMER', `52')